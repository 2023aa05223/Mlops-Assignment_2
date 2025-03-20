import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import H2O for AutoML
import h2o
from h2o.automl import H2OAutoML

# Optuna for hyperparameter optimization
import optuna
from sklearn.ensemble import RandomForestClassifier

# Load the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Reshape the data from 28x28 images to a flat vector
X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Create a validation set from the training data
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_val_scaled = scaler.transform(X_val.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Validation data shape: {X_val_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")

# Fashion MNIST class names for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 1. Model Selection using H2O AutoML
print("\nInitializing H2O...")
h2o.init()

# Convert data to H2O frames
# For better memory management, use a subset of the data
sample_size = 10000  # Adjust based on available memory

# Create pandas DataFrames for H2O
train_df = pd.DataFrame(X_train_scaled[:sample_size], columns=[f'pixel_{i}' for i in range(X_train_scaled.shape[1])])
train_df['label'] = y_train[:sample_size]

val_df = pd.DataFrame(X_val_scaled, columns=[f'pixel_{i}' for i in range(X_val_scaled.shape[1])])
val_df['label'] = y_val

# Convert to H2O frames
train_h2o = h2o.H2OFrame(train_df)
val_h2o = h2o.H2OFrame(val_df)

# Specify the response column
y_col = 'label'
x_cols = train_h2o.columns
x_cols.remove(y_col)

# Convert the response column to categorical (factor)
train_h2o[y_col] = train_h2o[y_col].asfactor()
val_h2o[y_col] = val_h2o[y_col].asfactor()

print("\nRunning H2O AutoML...")
# Initialize and run H2O AutoML
automl = H2OAutoML(
    max_runtime_secs=600,  # 10 minutes
    max_models=20,
    seed=42,
    sort_metric='logloss'
)

automl.train(x=x_cols, y=y_col, training_frame=train_h2o, validation_frame=val_h2o)

# Get the leaderboard
lb = automl.leaderboard
print("\nH2O AutoML Leaderboard:")
print(lb.head(10))

# Get the best model
best_model = automl.leader
print(f"\nBest model: {best_model}")

# Save the best model from H2O
model_path = h2o.save_model(model=best_model, path="./", force=True)
print(f"H2O model saved to: {model_path}")

# Extract the best model type for sklearn implementation
model_type = best_model._model_json["output"]["model_category"]
print(f"Best model type: {model_type}")

# 2. Hyperparameter Optimization with Optuna
print("\nOptimizing hyperparameters with Optuna for a similar model in sklearn...")

# Define an objective function for Optuna using RandomForest
# (RandomForest is often among the best models for this type of task)
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 4, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
    }
    
    # Create model with the suggested parameters
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Adjust n_trials based on time constraints

# Get best parameters and create the optimized model
best_params = study.best_params
print(f"Best parameters: {best_params}")

# Train the final model with the best parameters
final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train_scaled, y_train)

# Evaluate on the test set
y_pred = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal model test accuracy: {test_accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('fashion_mnist_confusion_matrix.png')
print("Confusion matrix saved as 'fashion_mnist_confusion_matrix.png'")

# Save the final model
import joblib
joblib.dump(final_model, 'fashion_mnist_optimized_model.pkl')
print("Model saved as 'fashion_mnist_optimized_model.pkl'")

# Shutdown H2O
h2o.shutdown(prompt=False)
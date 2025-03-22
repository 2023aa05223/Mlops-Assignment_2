import datetime
import os
import pickle
import h2o
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from h2o.automl import H2OAutoML
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# Initialize H2O
h2o.init()

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Flatten images and normalize
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Convert to Pandas DataFrame
df_train = pd.DataFrame(x_train)
df_train['label'] = y_train
df_test = pd.DataFrame(x_test)
df_test['label'] = y_test

# Reduce dataset size for faster execution
df_train = df_train.sample(frac=0.3, random_state=42)  # Use 30% of data
df_test = df_test.sample(frac=0.3, random_state=42)  # Use 30% of test data

# Convert to H2O Frame
hf_train = h2o.H2OFrame(df_train)
hf_test = h2o.H2OFrame(df_test)

# Define response and predictors
response = 'label'
predictors = df_train.columns[:-1].tolist()
hf_train[response] = hf_train[response].asfactor()
hf_test[response] = hf_test[response].asfactor()

# Run H2O AutoML with reduced model count and runtime
aml = H2OAutoML(max_models=5, max_runtime_secs=600, seed=42)
aml.train(x=predictors, y=response, training_frame=hf_train)

# Create output directory
output_dir = "reports"
os.makedirs(output_dir, exist_ok=True)
file_path = output_dir+"/automl_results.csv"

# Get AutoML leaderboard and save results
aml_leaderboard = aml.leaderboard.as_data_frame()
aml_leaderboard.to_csv(file_path, index=False)
print("AutoML Model Comparison:")
print(aml_leaderboard)

# Get best model
best_model = aml.leader
print(f"\nBest model: {best_model}")

# Save the best model from H2O
model_path = h2o.save_model(model=best_model, path="models", force=True)
print(f"H2O model saved to: {model_path}")

# Extract the best model type for sklearn implementation
model_type = best_model._model_json["output"]["model_category"]
print(f"Best model type: {model_type}")

final_model = best_model

# Define Optuna objective for hyperparameter tuning using the best model
def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learn_rate", 0.001, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    ntrees = trial.suggest_int("ntrees", 50, 300, step=50)  # Reduced max trees
    min_rows = trial.suggest_int("min_rows", 1, 10)
    sample_rate = trial.suggest_float("sample_rate", 0.5, 1.0)
    col_sample_rate = trial.suggest_float("col_sample_rate", 0.5, 1.0)
    stopping_rounds = trial.suggest_int("stopping_rounds", 5, 20)
    
    # Clone best model and apply new hyperparameters
    model = best_model
    model.set_params(
        learn_rate=learning_rate, max_depth=max_depth, ntrees=ntrees,
        min_rows=min_rows, sample_rate=sample_rate, col_sample_rate=col_sample_rate,
        stopping_rounds=stopping_rounds
    )
    
    # Train model
    model.train(x=predictors, y=response, training_frame=hf_train)
    
    # Predict and evaluate
    preds = model.predict(hf_test).as_data_frame()['predict'].astype(int)
    accuracy = accuracy_score(y_test, preds)
    final_model = model
    return accuracy

# Run hyperparameter optimization with fewer trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Reduced from 20 to 10

# Save hyperparameter tuning logs
with open("hyperparameter_tuning_logs.txt", "w") as f:
    for trial in study.trials:
        f.write(f"Trial {trial.number}: {trial.params}, Accuracy: {trial.value}\n")

# Print best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Justification for chosen model and hyperparameters
best_model_name = best_model.model_id
justification = f"The best model selected by H2O AutoML is {best_model_name}, based on performance on the validation data. The chosen hyperparameters from Optuna tuning (learning rate: {study.best_params['learn_rate']}, max depth: {study.best_params['max_depth']}, trees: {study.best_params['ntrees']}, min rows: {study.best_params['min_rows']}, sample rate: {study.best_params['sample_rate']}, col sample rate: {study.best_params['col_sample_rate']}, stopping rounds: {study.best_params['stopping_rounds']}, balance classes: {study.best_params['balance_classes']}) improve accuracy."

with open("model_justification.txt", "w") as f:
    f.write(justification)

print(justification)

# Shutdown H2O
h2o.shutdown(prompt=False)

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

# Define a threshold for PSI
PSI_THRESHOLD = 0.1

# Compute PSI to detect drift
psi_value = calculate_psi(y_train, y_test)
print(f"PSI Value: {psi_value}")

if psi_value > PSI_THRESHOLD:
    print("Drift detected. Retraining the model...")
    
    # Retrain the model using H2O AutoML
    aml_retrain = H2OAutoML(max_models=5, max_runtime_secs=600, seed=42)
    aml_retrain.train(x=predictors, y=response, training_frame=hf_train)
    
    # Get the best model from retraining
    best_model_retrain = aml_retrain.leader
    print(f"Retrained Best Model: {best_model_retrain}")
    
    # Save the retrained model
    retrain_model_path = h2o.save_model(model=best_model_retrain, path="models", force=True)
    print(f"Retrained H2O model saved to: {retrain_model_path}")
    
    # Update the final model
    final_model = best_model_retrain
else:
    print("No significant drift detected. Skipping retraining.")

# MLflow Tracking
mlflow.set_experiment("FashionMNIST_Tracking")
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
# Start an MLflow run
with mlflow.start_run(run_name=run_name) as mlflow_run:
    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)
    
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("purpose", "AutoML and Hyperparameter Tuning")
    mlflow.log_params(best_params)
    
    # Evaluate model
    y_pred = np.argmax(final_model.predict(x_test), axis=1)
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    mlflow.log_metric("PSI_value", psi_value)
    
    # Infer the model signature
    signature = infer_signature(
        x_train, final_model.predict(x_train)
    )
    
with open('model.pkl', 'wb') as model_file:
    pickle.dump(final_model, model_file)
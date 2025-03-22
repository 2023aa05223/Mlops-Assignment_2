import logging
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
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import time

# Constants
OUTPUT_DIR = "reports"
MODELS_DIR = "models"
PSI_THRESHOLD = 0.1
H2O_MAX_MODELS = 5
H2O_MAX_RUNTIME_SECS = 600
OPTUNA_TRIALS = 3  # Reduce the number of trials for faster execution

# Optuna Hyperparameter Ranges
OPTUNA_LEARNING_RATE_MIN = 0.01
OPTUNA_LEARNING_RATE_MAX = 0.1
OPTUNA_MAX_DEPTH_MIN = 3
OPTUNA_MAX_DEPTH_MAX = 6
OPTUNA_NTREES_MIN = 50
OPTUNA_NTREES_MAX = 150
OPTUNA_NTREES_STEP = 50
OPTUNA_MIN_ROWS_MIN = 1
OPTUNA_MIN_ROWS_MAX = 10
OPTUNA_SAMPLE_RATE_MIN = 0.5
OPTUNA_SAMPLE_RATE_MAX = 1.0
OPTUNA_COL_SAMPLE_RATE_MIN = 0.5
OPTUNA_COL_SAMPLE_RATE_MAX = 1.0
OPTUNA_STOPPING_ROUNDS_MIN = 5
OPTUNA_STOPPING_ROUNDS_MAX = 10

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_h2o():
    """Initialize the H2O environment."""
    h2o.init()

def load_and_preprocess_data():
    """Load and preprocess the Fashion MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Reduce dataset size for faster execution
    sample_indices_train = np.random.choice(len(x_train), size=int(len(x_train) * 0.3), replace=False)
    sample_indices_test = np.random.choice(len(x_test), size=int(len(x_test) * 0.3), replace=False)

    x_train = x_train[sample_indices_train]
    y_train = y_train[sample_indices_train]
    x_test = x_test[sample_indices_test]
    y_test = y_test[sample_indices_test]

    return x_train, y_train, x_test, y_test

def convert_to_h2o_frame(x_train, y_train, x_test, y_test):
    """Convert data to H2O Frames."""
    df_train = pd.DataFrame(x_train)
    df_train['label'] = y_train
    df_test = pd.DataFrame(x_test)
    df_test['label'] = y_test

    hf_train = h2o.H2OFrame(df_train)
    hf_test = h2o.H2OFrame(df_test)

    hf_train['label'] = hf_train['label'].asfactor()
    hf_test['label'] = hf_test['label'].asfactor()

    return hf_train, hf_test, df_train.columns[:-1].tolist()

def run_h2o_automl(hf_train, predictors, response):
    """Run H2O AutoML and return the best model."""
    aml = H2OAutoML(max_models=H2O_MAX_MODELS, max_runtime_secs=H2O_MAX_RUNTIME_SECS, seed=42)
    aml.train(x=predictors, y=response, training_frame=hf_train)
    return aml

def save_automl_results(aml, output_dir):
    """Save AutoML leaderboard results."""
    os.makedirs(output_dir, exist_ok=True)
    leaderboard_path = os.path.join(output_dir, "automl_results.csv")
    aml.leaderboard.as_data_frame().to_csv(leaderboard_path, index=False)
    logger.info("AutoML Model Comparison saved to %s", leaderboard_path)

def save_model(model, models_dir):
    """Save the best model."""
    os.makedirs(models_dir, exist_ok=True)
    try:
        model_path = h2o.save_model(model=model, path=models_dir, force=True)
        logger.info("H2O model saved to: %s", model_path)
    except Exception as e:
        logger.error("Failed to save H2O model: %s", e)

def save_model_as_pickle(model, output_dir):
    """Save the best model as a pickle file."""
    pickle_path = os.path.join(output_dir, "best_model.pkl")
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Best model saved as pickle to: %s", pickle_path)
    except Exception as e:
        logger.error("Failed to save model as pickle: %s", e)

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) to detect data drift.
    """
    def get_bucket_values(data, buckets):
        percentiles = np.linspace(0, 100, buckets + 1)
        bucket_edges = np.percentile(data, percentiles)
        return np.histogram(data, bins=bucket_edges)[0] / len(data)
    
    expected_dist = get_bucket_values(expected, buckets)
    actual_dist = get_bucket_values(actual, buckets)
    psi_values = (expected_dist - actual_dist) * np.log((expected_dist + 1e-8) / (actual_dist + 1e-8))
    return np.sum(psi_values)

def detect_and_handle_drift(hf_train, hf_test, predictors, response, y_train, y_test):
    """Detect data drift using PSI and retrain the model if necessary."""
    psi_value = calculate_psi(y_train, y_test)
    logger.info("PSI Value: %f", psi_value)

    if psi_value > PSI_THRESHOLD:
        logger.warning("Drift detected. Retraining the model...")
        aml_retrain = H2OAutoML(max_models=H2O_MAX_MODELS, max_runtime_secs=H2O_MAX_RUNTIME_SECS, seed=42)
        aml_retrain.train(x=predictors, y=response, training_frame=hf_train)
        save_model(aml_retrain.leader, MODELS_DIR)
    else:
        logger.info("No significant drift detected. Skipping retraining.")

    return psi_value

def run_optuna_tuning(best_model, hf_train, hf_test, predictors, response, y_test):
    """Run Optuna hyperparameter tuning and update the best model."""
    start_time = time.time()  # Start timer

    def objective(trial):
        learning_rate = trial.suggest_float("learn_rate", OPTUNA_LEARNING_RATE_MIN, OPTUNA_LEARNING_RATE_MAX, log=True)
        max_depth = trial.suggest_int("max_depth", OPTUNA_MAX_DEPTH_MIN, OPTUNA_MAX_DEPTH_MAX)
        ntrees = trial.suggest_int("ntrees", OPTUNA_NTREES_MIN, OPTUNA_NTREES_MAX, step=OPTUNA_NTREES_STEP)
        min_rows = trial.suggest_int("min_rows", OPTUNA_MIN_ROWS_MIN, OPTUNA_MIN_ROWS_MAX)
        sample_rate = trial.suggest_float("sample_rate", OPTUNA_SAMPLE_RATE_MIN, OPTUNA_SAMPLE_RATE_MAX)
        col_sample_rate = trial.suggest_float("col_sample_rate", OPTUNA_COL_SAMPLE_RATE_MIN, OPTUNA_COL_SAMPLE_RATE_MAX)
        stopping_rounds = trial.suggest_int("stopping_rounds", OPTUNA_STOPPING_ROUNDS_MIN, OPTUNA_STOPPING_ROUNDS_MAX)

        model = best_model
        model.set_params(
            learn_rate=learning_rate, max_depth=max_depth, ntrees=ntrees,
            min_rows=min_rows, sample_rate=sample_rate, col_sample_rate=col_sample_rate,
            stopping_rounds=stopping_rounds
        )
        model.train(x=predictors, y=response, training_frame=hf_train)
        preds = model.predict(hf_test).as_data_frame()['predict'].astype(int)
        return accuracy_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=2)

    end_time = time.time()  # End timer
    logger.info("Optuna tuning completed in %.2f seconds", end_time - start_time)

    logger.info("Best Hyperparameters: %s", study.best_params)
    logger.info("Best Accuracy from Optuna: %f", study.best_value)

    # Update the best model with the best hyperparameters
    best_params = study.best_params
    best_model.set_params(
        learn_rate=best_params["learn_rate"],
        max_depth=best_params["max_depth"],
        ntrees=best_params["ntrees"],
        min_rows=best_params["min_rows"],
        sample_rate=best_params["sample_rate"],
        col_sample_rate=best_params["col_sample_rate"],
        stopping_rounds=best_params["stopping_rounds"]
    )
    best_model.train(x=predictors, y=response, training_frame=hf_train)

    return best_model, study

def save_hyperparameter_logs(study, output_dir):
    """Save hyperparameter tuning logs."""
    tuning_log_path = os.path.join(output_dir, "hyperparameter_tuning_logs.txt")
    try:
        with open(tuning_log_path, "w") as f:
            for trial in study.trials:
                f.write(f"Trial {trial.number}: {trial.params}, Accuracy: {trial.value}\n")
        logger.info("Hyperparameter tuning logs saved to %s", tuning_log_path)
    except Exception as e:
        logger.error("Failed to save hyperparameter tuning logs: %s", e)

def log_mlflow_metrics(best_model, study, psi_value, x_train, x_test, y_test):
    """Log metrics and parameters to MLflow."""
    mlflow.set_experiment("FashionMNIST_Tracking")
    with mlflow.start_run(run_name=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) as run:
        run_id = run.info.run_id
        logger.info("MLflow Run ID: %s", run_id)

        # Log parameters and metrics
        mlflow.log_params(study.best_params)
        mlflow.log_metric("Best_Accuracy", study.best_value)
        mlflow.log_metric("PSI_Value", psi_value)  # Added PSI value as a metric

        # Save the model as an artifact
        pickle_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
        mlflow.log_artifact(pickle_path)

def main():
    initialize_h2o()
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    hf_train, hf_test, predictors = convert_to_h2o_frame(x_train, y_train, x_test, y_test)
    response = 'label'

    # Run H2O AutoML
    aml = run_h2o_automl(hf_train, predictors, response)
    save_automl_results(aml, OUTPUT_DIR)
    best_model = aml.leader

    # Detect drift and calculate PSI value
    psi_value = detect_and_handle_drift(hf_train, hf_test, predictors, response, y_train, y_test)

    # Run Optuna tuning and update the best model
    best_model, study = run_optuna_tuning(best_model, hf_train, hf_test, predictors, response, y_test)
    save_hyperparameter_logs(study, OUTPUT_DIR)

    # Save the updated model after Optuna tuning
    save_model(best_model, MODELS_DIR)
    save_model_as_pickle(best_model, OUTPUT_DIR)

    # Log metrics to MLflow, including PSI value
    log_mlflow_metrics(best_model, study, psi_value, x_train, x_test, y_test)

    try:
        h2o.shutdown(prompt=False)
        logger.info("H2O shutdown successfully.")
    except Exception as e:
        logger.error("Failed to shutdown H2O: %s", e)

if __name__ == "__main__":
    main()
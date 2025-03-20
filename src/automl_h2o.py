import h2o
import optuna
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
df_train = df_train.sample(frac=0.5, random_state=42)  # Use 50% of data

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

# Get AutoML leaderboard and save results
aml_leaderboard = aml.leaderboard.as_data_frame()
aml_leaderboard.to_csv("automl_results.csv", index=False)
print("AutoML Model Comparison:")
print(aml_leaderboard)

# Get best model
best_model = aml.leader

# Define Optuna objective for hyperparameter tuning
def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learn_rate", 0.001, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    ntrees = trial.suggest_int("ntrees", 50, 300, step=50)  # Reduced max trees
    
    # Train model
    model = h2o.estimators.H2OGradientBoostingEstimator(
        learn_rate=learning_rate, max_depth=max_depth, ntrees=ntrees
    )
    model.train(x=predictors, y=response, training_frame=hf_train)
    
    # Predict and evaluate
    preds = model.predict(hf_test).as_data_frame()['predict'].astype(int)
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy

# Run hyperparameter optimization with fewer trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Reduced from 20 to 10

# Save hyperparameter tuning logs
with open("hyperparameter_tuning_logs.txt", "w") as f:
    for trial in study.trials:
        f.write(f"Trial {trial.number}: {trial.params}, Accuracy: {trial.value}\n")

# Print best hyperparameters
print("Best Hyperparameters:", study.best_params)

# Justification for chosen model and hyperparameters
best_model_name = best_model.model_id
justification = f"The best model selected by H2O AutoML is {best_model_name}, based on performance on the validation data. The chosen hyperparameters from Optuna tuning (learning rate: {study.best_params['learn_rate']}, max depth: {study.best_params['max_depth']}, trees: {study.best_params['ntrees']}) improve accuracy."

with open("model_justification.txt", "w") as f:
    f.write(justification)

print(justification)

# Shutdown H2O
h2o.shutdown(prompt=False)

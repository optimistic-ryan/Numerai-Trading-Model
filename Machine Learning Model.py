# numerapi, for facilitating data download and predictions uploading
# catboost, for modeling and making predictions
!pip install numerapi
!pip install catboost

# Import necessary libraries
import os
import gc
import csv
import glob
import time
from pathlib import Path

import numerapi

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso

# Initialize the NumerAPI to communicate with Numerai's servers
napi = numerapi.NumerAPI(verbosity="info")
# Download the latest data from Numerai
napi.download_current_dataset(unzip=True)

# Get the number of the current round
current_ds = napi.get_current_round()
# Form the name of the latest round's data directory
latest_round = os.path.join('numerai_dataset_'+str(current_ds))
# Define the name of the tournament and the target variable
tournament_name = "nomi"
target_name = f"target"
# Define the name of the column to store the predictions in
prediction_name = f"prediction"

# Define constants for payout calculation
benchmark = 0
band = 0.25

# Function to calculate spearman correlation, which is used by Numerai to score the model's performance
def score(df):
    return np.corrcoef(
        df[target_name],
        df[prediction_name].rank(pct=True, method="first")
    )[0, 1]

# Function to calculate payout based on the model's score
def payout(scores):
    return ((scores - benchmark) / band).clip(lower=-1, upper=1)

# Function to read in the CSV files from Numerai, specifying float16 for numeric columns to save memory
def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))
        dtypes = {x: np.float16 for x in column_names if
                x.startswith(('feature', 'target'))}
    return pd.read_csv(file_path, dtype=dtypes)

# Read in the training and tournament data
training_data = read_csv(os.path.join(latest_round, "numerai_training_data.csv")).set_index("id")
tournament_data = read_csv(os.path.join(latest_round, "numerai_tournament_data.csv")).set_index("id")

# Read in the example predictions as a reference
example_preds = read_csv(os.path.join(latest_round, "example_predictions.csv"))

# Extract the validation data from the tournament data
validation_data = tournament_data[tournament_data.data_type == "validation"]

# Get the names of the feature columns
feature_names = [f for f in training_data.columns if f.startswith("feature")]

# Function to add interaction and polynomial features and impute missing values
def engineer_features(df):
    # Create interaction features
    interaction_cols = [('feature_1', 'feature_2'), ('feature_1', 'feature_3'), ('feature_2', 'feature_3')]
    for col1, col2 in interaction_cols:
        if col1 in df.columns and col2 in df.columns:
            df[col1] = pd.to_numeric(df[col1], errors='coerce')
            df[col2] = pd.to_numeric(df[col2], errors='coerce')
            df[f'{col1}_{col2}'] = df[col1] * df[col2]
    
    # Create polynomial features
    poly_cols = ['feature_1', 'feature_2', 'feature_3']
    for col in poly_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}^2'] = df[col] ** 2
            df[f'{col}^3'] = df[col] ** 3
        
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df[feature_names] = imputer.fit_transform(df[feature_names])
    
    return df

# Apply feature engineering to all datasets
training_data = engineer_features(training_data)
validation_data = engineer_features(validation_data)
tournament_data = engineer_features(tournament_data)

# Set up a CatBoostRegressor with GPU acceleration
params = {'task_type': 'GPU'}
estimator = CatBoostRegressor(**params)

# Define a grid of hyperparameters to search over
grid = {'learning_rate': [0.1, 0.01, 0.001],'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(estimator, grid, cv=3, n_jobs=-1, verbose=3)

# Perform the grid search to find the best hyperparameters
grid_search.fit(training_data[feature_names].astype(np.float32), training_data[target_name].astype(np.float32))

# Extract the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Print the best hyperparameters
print(f"Best hyperparameters: {best_params}")

# Function to generate ensemble predictions from a list of models
def ensemble_predict(models, X):
    return np.mean([model.predict(X) for model in models], axis=0)

# Train a CatBoost model with the best hyperparameters and generate predictions for the training and tournament data
training_data[prediction_name] = ensemble_predict(models, training_data[feature_names].astype(np.float32))
tournament_data[prediction_name] = ensemble_predict(models, tournament_data[feature_names].astype(np.float32))

# Apply L1 regularization to the models
reg_params = {'alpha': 0.01}
for model in models:
    model.add_regularizer(Lasso(**reg_params))

# Evaluate the model using 5-fold cross-validation
scores = cross_val_score(estimator, training_data[feature_names].astype(np.float32), training_data[target_name].astype(np.float32), cv=5, scoring=score)
print(f"On training the correlation has mean {scores.mean()} and std {scores.std()}")
print(f"On training the average per-era payout is {payout(scores).mean()}")

# Impute missing values using the mean of each feature
imputer = SimpleImputer(strategy='mean')
training_data[feature_names] = imputer.fit_transform(training_data[feature_names])
validation_data[feature_names] = imputer.transform(validation_data[feature_names])
tournament_data[feature_names] = imputer.transform(tournament_data[feature_names])

# Use PCA to reduce the number of features to 100
pca = PCA(n_components=100)
training_data[feature_names] = pca.fit_transform(training_data[feature_names])
validation_data[feature_names] = pca.transform(validation_data[feature_names])
tournament_data[feature_names] = pca.transform(tournament_data[feature_names])

# Train a LightGBM model and generate predictions for the training and tournament data
from lightgbm import LGBMRegressor

lgbm_params = {'objective': 'regression', 'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.01, 'reg_lambda': 0.01, 'random_state': 42}

lgbm = LGBMRegressor(**lgbm_params)
lgbm.fit(training_data[feature_names], training_data[target_name])

training_data['lgbm_prediction'] = lgbm.predict(training_data[feature_names])
tournament_data['lgbm_prediction'] = lgbm.predict(tournament_data[feature_names])

# Evaluate the LightGBM model using 5-fold cross-validation
lgbm_scores = cross_val_score(lgbm, training_data[feature_names], training_data[target_name], cv=5, scoring=score)
print(f"On training the correlation has mean {lgbm_scores.mean()} and std {lgbm_scores.std()}")
print(f"On training the average per-era payout is {payout(lgbm_scores).mean()}")

# Combine the predictions from the CatBoost and LightGBM models
tournament_data[prediction_name] = (tournament_data[prediction_name] + tournament_data['lgbm_prediction']) / 2

# Save the final predictions to a CSV file
tournament_data[prediction_name].to_csv(f"{tournament_name}_{current_ds}_submission.csv")

# Check the per-era correlations on the validation data
validation_data = tournament_data[tournament_data.data_type == "validation"]
validation_correlations = validation_data.groupby("era").apply(score)
print(f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

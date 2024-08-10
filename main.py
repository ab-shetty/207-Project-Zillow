import argparse
import sys
import json
import random

import tensorflow as tf
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import partial
import optuna

from models.load_data import load_mlr_data, load_xgb_data
from models.mlr import process_mlr_data, build_mlr_model, generate_mlr_prediction
from models.xgb import process_xgb_data, train_tune_xgb_model, generate_xgb_prediction
from models.nn import process_nn_data, nn_train_val, objective_nn, build_model, read_test_data, create_clean_test
from models.combo import generate_combo_prediction

# filter warnings
import warnings
warnings.filterwarnings('ignore')

# Create submission file
def save_submission(p_test):
    sub = pd.read_csv('./data/sample_submission.csv')
    for c in sub.columns[sub.columns != 'ParcelId']:
        sub[c] = p_test
    print('Save submission')
    sub.to_csv('submission.csv', index=False, float_format='%.4f')

# Function placeholders

base_path = './data/'

def run_linear_regression():
    print("Running Linear Regression model and saving submission file...")
    ## Load Data ##
    df_properties, sample, df_logs, df_train = load_mlr_data(base_path)

    ## MLR data processing ##
    X_train, X_val, y_train, y_val, data = process_mlr_data(df_train)

    # Standardizing all features in X_train, X_val, and X_test
    X_train_std = (X_train-X_train.mean())/X_train.std()
    X_val_std = (X_val-X_train.mean())/X_train.std()

    # Standardizing Y_train, Y_val, and Y_test
    y_train_std = (y_train-y_train.mean())/y_train.std()
    y_val_std = (y_val-y_train.mean())/y_train.std()

    ## Build and Train Model ##
    # Build and compile test_model
    test_mlr_model = build_mlr_model(
        num_features=X_train.shape[1], learning_rate=0.0007)

    # Fit test model
    test_num_epochs = 10
    test_train_tf = test_mlr_model.fit(x=X_train_std, y=y_train_std, epochs=test_num_epochs, verbose=0,
                                       validation_data=(X_val_std, y_val_std))

    num_missing, average_preds, test_preds = generate_mlr_prediction(
        test_mlr_model, df_properties, sample, df_logs, df_train, X_train, X_val, y_train, y_val, data)

    # Replacing missing predictions with average of test data predictions
    for i in range(num_missing):
        test_preds = np.append(test_preds, average_preds)
    save_submission(test_preds)


def run_neural_network(nn_only=True):
    print("Running Neural Network model...")

    df_properties, sample, df_logs, df_all = load_mlr_data(base_path)
    X_train, X_val, Y_train, Y_val, X, Y = process_nn_data(df_all)

    X_train_std, X_val_std, Y_train_std, Y_val_std, y_mean, y_std = nn_train_val(
        X_train, X_val, Y_train, Y_val)

    train_x = {
        'bedroomcnt': X_train_std[['bedroomcnt']],
        'roomcnt': X_train_std[['roomcnt']],
        'bathroomcnt': X_train_std[['bathroomcnt']],
        'taxamount': X_train_std[['taxamount']],
        'landtaxvaluedollarcnt': X_train_std[['landtaxvaluedollarcnt']],
        'taxvaluedollarcnt': X_train_std[['taxvaluedollarcnt']],
        'structuretaxvaluedollarcnt': X_train_std[['structuretaxvaluedollarcnt']],
        'latitude': X_train_std[['latitude']],
        'longitude': X_train_std[['longitude']],
        'year': X_train_std[['year']],
        'month': X_train_std[['month']],
        'weekday': X_train_std[['weekday']],
        'lotsizesquarefeet': X_train_std[['lotsizesquarefeet']],
        'calculatedfinishedsquarefeet': X_train_std[['calculatedfinishedsquarefeet']],
        'yearbuilt': X_train_std[['yearbuilt']], }

    val_x = {
        'bedroomcnt': X_val_std[['bedroomcnt']],
        'roomcnt': X_val_std[['roomcnt']],
        'bathroomcnt': X_val_std[['bathroomcnt']],
        'taxamount': X_val_std[['taxamount']],
        'landtaxvaluedollarcnt': X_val_std[['landtaxvaluedollarcnt']],
        'taxvaluedollarcnt': X_val_std[['taxvaluedollarcnt']],
        'structuretaxvaluedollarcnt': X_val_std[['structuretaxvaluedollarcnt']],
        'latitude': X_val_std[['latitude']],
        'longitude': X_val_std[['longitude']],
        'year': X_val_std[['year']],
        'month': X_val_std[['month']],
        'weekday': X_val_std[['weekday']],
        'lotsizesquarefeet': X_val_std[['lotsizesquarefeet']],
        'calculatedfinishedsquarefeet': X_val_std[['calculatedfinishedsquarefeet']],
        'yearbuilt': X_val_std[['yearbuilt']],
    }

    # Create a study object and optimize the objective function
    objective_partial = partial(
        objective_nn, train_x=train_x, Y_train_std=Y_train_std, val_x=val_x, Y_val_std=Y_val_std)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_partial, n_trials=5, timeout=900)

    # Extracting the best parameters from the study
    best_params = study.best_params

    # Unpacking the best parameters into individual variables
    lr = best_params["lr"]
    resolution_in_degrees = best_params["resolution_in_degrees"]
    epochs = best_params["epochs"]
    batch = best_params["batch"]

    model = build_model(lr, resolution_in_degrees)
    random.seed(42)
    tf.random.set_seed(1234)

    # Fit model
    model.fit(
            x=train_x,
            y=Y_train_std,
            epochs=epochs,
            batch_size=batch,
            validation_data=(val_x, Y_val_std)
        )

    df_test, sample = read_test_data()
    X_test_std = create_clean_test(df_test, X_train)

    # Generate neural network predictions for all parcel IDs
    preds = model.predict({
            'bedroomcnt': X_test_std[['bedroomcnt']],
            'roomcnt': X_test_std[['roomcnt']],
            'bathroomcnt': X_test_std[['bathroomcnt']],
            'taxamount': X_test_std[['taxamount']],
            'landtaxvaluedollarcnt': X_test_std[['landtaxvaluedollarcnt']],
            'taxvaluedollarcnt': X_test_std[['taxvaluedollarcnt']],
            'structuretaxvaluedollarcnt': X_test_std[['structuretaxvaluedollarcnt']],
            'latitude': X_test_std[['latitude']],
            'longitude': X_test_std[['longitude']],
            'year': X_test_std[['year']],
            'month': X_test_std[['month']],
            'weekday': X_test_std[['weekday']],
            'lotsizesquarefeet': X_test_std[['lotsizesquarefeet']],
            'calculatedfinishedsquarefeet': X_test_std[['calculatedfinishedsquarefeet']],
            'yearbuilt': X_test_std[['yearbuilt']],
    })

    s = ((preds[:, 0]*y_std) + y_mean)

    if not nn_only:
        train_preds = model.predict(train_x)
        val_preds = model.predict(val_x)

        # Convert to orginal scale from standardized scale
        train_preds = (train_preds[:,0]*Y_train.std()) + Y_train.mean()
        val_preds = (val_preds[:,0]*Y_train.std()) + Y_train.mean()
        return s, Y_train, Y_val, train_preds, val_preds
    
    save_submission(s)


# ANOTHER ONE
def run_xgboost():
    print("Training XGBoost model...")
    ## Load Data ##
    train, prop, sample = load_xgb_data(base_path)

    ## XGBoost processing ##
    # Build XGBoost DMatrix objects for efficient processing
    x_train, y_train, x_valid, y_valid, train_columns = process_xgb_data(
        train, prop)
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    study = train_tune_xgb_model(d_train, d_valid)

    # Train Model on Best Parameters
    params = study.best_params
    #params = {'eta': 0.022382530987582482, 'max_depth': 5, 'subsample': 0.6561243067188417, 'colsample_bytree': 0.6775779485416246, 'n_estimators': 640, 'lambda': 0.9942582014915469, 'alpha': 7.97377143057426e-08}
    params['objective'] = 'reg:linear'
    params['eval_metric'] = 'mae'

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 10000, watchlist,
                    early_stopping_rounds=100, verbose_eval=10)

    p_test = generate_xgb_prediction(clf, prop, sample, train_columns)

    save_submission(p_test)


# ANOTHER FUNCTIONS

def run_combination_model():
    print("Running Combination Model (NN + XGBoost)...")

    s, Y_train, Y_val, train_preds, val_preds = run_neural_network(False)

    # Create XGBoost matrices
    #d_train = xgb.DMatrix(train_preds, label = Y_train)
    #d_valid = xgb.DMatrix(val_preds, label = Y_val)
    d_train = xgb.DMatrix(train_preds.reshape(-1, 1), label=Y_train)
    d_valid = xgb.DMatrix(val_preds.reshape(-1, 1), label=Y_val)

    # Train XGBoost model
    study = train_tune_xgb_model(d_train, d_valid)

    # Train Model on Best Parameters
    params = study.best_params
    params['objective'] = 'reg:linear'
    params['eval_metric'] = 'mae'

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 10000, watchlist,
                    early_stopping_rounds=100, verbose_eval=10)

    p_test = generate_combo_prediction(clf, preds=s) # For XGBoost
    save_submission(p_test)

def load_pretrained_model_and_predict(model_name, feature_data):
    # Define the path to the pretrained models
    model_paths = {
        'linear_regression': './pretrained_models/linear_regression.keras',
        'neural_network': './pretrained_models/nn.keras',
        'xgboost': './pretrained_models/xgboost.model',
        'combination': './pretrained_models/nn_xgboost_combination.keras'
    }

    if model_name not in model_paths:
        print(
            f"Model {model_name} not found. Please choose from {list(model_paths.keys())}")
        sys.exit(1)

    model_path = model_paths[model_name]

    print(f"Loading pretrained model from {model_path}...")

    # Load the model (assuming Keras models, modify as needed for XGBoost or other formats)
    if model_name == 'xgboost':
        # Add XGBoost model loading logic here
        pass  # Placeholder for XGBoost model loading
    else:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

    # Convert feature data into the format expected by the model
    predictions = []
    for features in feature_data:
        feature_values = [features[key] for key in features]
        # Convert to batch format (list of lists)
        feature_values = [feature_values]
        prediction = model.predict(feature_values)
        predictions.append(prediction)

    print("Predictions generated:")
    for i, prediction in enumerate(predictions):
        print(f"Input {i+1}: {feature_data[i]} => Prediction: {prediction}")


def setup_local():
    print("Setting up for local environment...")
    # Replace with actual data loading logic
    data_path = "./data/"
    print(f"Data should be located in {data_path}")
    return data_path


def setup_colab():
    print("Setting up for Google Colab environment...")
    # from google.colab import drive
    # drive.mount('/content/drive')
    data_path = input("Please enter the path to your data on Google Drive: ")
    print(f"Data path set to {data_path}")
    return data_path


def main():
    parser = argparse.ArgumentParser(
        description="Run the Zillow project pipeline.")

    # Add argument to select environment
    parser.add_argument('--env', choices=['local', 'colab'], required=True,
                        help="Choose the environment to run the pipeline: 'local' or 'colab'.")

    # Add argument to select model to train
    parser.add_argument('--model', choices=['linear_regression', 'neural_network', 'xgboost', 'combination'],
                        help="Choose the model to train: 'linear_regression', 'neural_network', 'xgboost', or 'combination'.")

    # Add argument to load a pretrained model and generate predictions
    parser.add_argument('--predict', metavar='model_name',
                        help="Name of the pretrained model to use for generating predictions.")

    # Add argument for feature data input
    parser.add_argument('--features', type=str,
                        help="Path to a JSON file containing the feature data for predictions.")

    args = parser.parse_args()

    # Set up environment
    if args.env == 'local':
        data_path = setup_local()
    elif args.env == 'colab':
        data_path = setup_colab()
    else:
        print("Invalid environment option.")
        sys.exit(1)

    # Train selected model
    if args.model:
        if args.model == 'linear_regression':
            run_linear_regression()
        elif args.model == 'neural_network':
            run_neural_network()
        elif args.model == 'xgboost':
            run_xgboost()
        elif args.model == 'combination':
            run_combination_model()
        else:
            print("Invalid model option.")
            sys.exit(1)

    # Load pretrained model and generate predictions
    # python main.py --env local --predict neural_network --features path/to/features.json

    if args.predict:
        if not args.features:
            print("Please provide the path to a JSON file containing the feature data for predictions using --features.")
            sys.exit(1)

        # Load feature data from JSON file
        with open(args.features, 'r') as f:
            feature_data = json.load(f)

        load_pretrained_model_and_predict(args.predict, feature_data)


if __name__ == "__main__":
    main()
    # run_combination_model()
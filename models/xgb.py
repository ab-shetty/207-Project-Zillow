import numpy as np
import optuna
import xgboost as xgb

from functools import partial


def process_xgb_data(train, prop):

    # Convert float64 columns to float32 to reduce memory usage
    for c, dtype in zip(prop.columns, prop.dtypes):
        if dtype == np.float64:
            prop[c] = prop[c].astype(np.float32)

    # Merge training data with property data
    df_train = train.merge(prop, how='left', on='parcelid')

    # Prepare features (X) and target variables (y) for training
    # Drop unnecessary columns
    x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate',
                            'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = df_train['logerror'].values

    # Store column names for later use
    train_columns = x_train.columns

    # Convert object (string) columns to boolean
    for c in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[c] = (x_train[c] == True)

    # Create training and validation dataset
    split = 80000
    x_train, y_train, x_valid, y_valid = x_train[:
                                                 split], y_train[:split], x_train[split:], y_train[split:]

    return x_train, y_train, x_valid, y_valid, train_columns


# Define the objective function
def objective(trial, d_train, d_valid):
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'eta': trial.suggest_float('eta', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
    }

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model
    model = xgb.train(params, d_train, num_boost_round=1000, evals=watchlist,
                      early_stopping_rounds=100, verbose_eval=False)

    # Return the best validation MAE
    val_mae = model.best_score
    return val_mae


def train_tune_xgb_model(d_train, d_valid):

    # Create a study object and optimize the objective function
    objective_partial = partial(objective, d_train=d_train, d_valid=d_valid)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_partial, n_trials=20, timeout=600)

    # Print the best parameters
    print(f"Best Parameters: {study.best_params}")

    return study


def generate_xgb_prediction(clf, prop, sample, train_columns, combo=False, preds=None):
    # Merge sample data with property data
    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop, on='parcelid', how='left')

    if combo:
        x_test = preds.reshape(-1, 1)
    else:
        # Convert object (string) columns to boolean
        x_test = df_test[train_columns]
        for c in x_test.dtypes[x_test.dtypes == object].index.values:
            x_test[c] = (x_test[c] == True)

    # Generate prediction
    d_test = xgb.DMatrix(x_test)
    p_test = clf.predict(d_test)

    return p_test

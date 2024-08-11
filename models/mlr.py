import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def process_mlr_data(df_train):
    # Dropping all rows where there is no longitude or latitidue data
    df_final = df_train[~df_train.regionidcounty.isnull()]

    # Want to use taxvaluedollarcnt, and since only 1 value is missing, will just drop that row
    df_final = df_final.drop(
        df_final[df_final.taxvaluedollarcnt.isnull()].index)

    # Getting column names of all columns w/o any missing values, and dropping parcelid
    selected = df_final.columns[df_final.apply(
        lambda c: c.isnull().sum() == 0)]

    # Getting selected columns
    data = df_final[selected]

    # Changing transactiondate to a datetime type
    data['transactiondate'] = pd.to_datetime(data['transactiondate'])

    # Extracting year and month based on the transaction date and setting them as
    # separate variables
    data['year'] = data['transactiondate'].dt.year

    # Setting train data to be all 2016 transactions
    X_train = data[data['year'] != 2017]
    y_train = X_train['logerror']

    # Dropping logerror (outcome variable), transactiondate (represented by month and day),
    # year and assessment year (since all observations in this subset have the same year value)
    X_train = X_train.drop(
        ['parcelid', 'logerror', 'transactiondate', 'year', 'assessmentyear', 'fips'], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, data


def build_mlr_model(num_features, learning_rate):
    """Build a TF linear regression model using Keras.

    Args:
      num_features: The number of input features.
      learning_rate: The desired learning rate for SGD.

    Returns:
      model: A tf.keras model (graph).
    """
    # This is not strictly necessary, but each time you build a model, TF adds
    # new nodes (rather than overwriting), so the colab session can end up
    # storing lots of copies of the graph when you only care about the most
    # recent. Also, as there is some randomness built into training with SGD,
    # setting a random seed ensures that results are the same on each identical
    # training run.
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    # Build a model using keras.Sequential. While this is intended for neural
    # networks (which may have multiple layers), we want just a single layer for
    # linear regression.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        units=1,        # output dim
        input_shape=[num_features],  # input dim
        use_bias=True,               # use a bias (intercept) param
        kernel_initializer=tf.ones_initializer,  # initialize params to 1
        bias_initializer=tf.ones_initializer,    # initialize bias to 1
    ))

    # We need to choose an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Finally, compile the model. This finalizes the graph for training.
    # We specify the loss and the optimizer above
    model.compile(
        optimizer=optimizer,
        loss='mae'
    )

    return model


def generate_mlr_prediction(test_mlr_model, df_properties, sample, df_logs, df_train, X_train, X_val, y_train, y_val, data):
    # Using sample submission to make test data for predictions for Kaggle submission
    X_test = data[data['parcelid'].isin(sample['ParcelId'])]
    sample['parcelid'] = sample['ParcelId']
    X_test = sample.merge(df_properties, on='parcelid', how='inner')

    # Getting number of rows with missing data
    num_missing = X_test[X_test[X_train.columns].isnull().any(axis=1)].shape[0]

    # Dropping rows with missing data
    X_test = X_test[X_train.columns].dropna()

    # Scaling the train and validation data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Creating predications based on the training and validation datasets
    train_preds = test_mlr_model.predict(X_train_scaled)
    val_preds = test_mlr_model.predict(X_val_scaled)

    # Printing out training and validation dataset mean absolute errors
    print("Train MAE:", mean_absolute_error(y_train, train_preds))
    print("Validation MAE:", mean_absolute_error(y_val, val_preds))

    # Scaling test data and making predictions
    X_test_scaled = scaler.transform(X_test)
    test_preds = test_mlr_model.predict(X_test_scaled)

    # Getting average of predictions
    average_preds = np.mean(test_preds)

    # Printing out mean absolute error between actual test data and test predictions
    y_test = df_logs['logerror']
    print("Test MAE:", mean_absolute_error(
        y_test, test_preds[len(test_preds)-len(y_test):]))

    return num_missing, average_preds, test_preds

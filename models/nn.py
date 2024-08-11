
import random

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers



def process_nn_data(df_all):
    # Remove some rows missing important data
    df = df_all.dropna(subset=['regionidcounty', 'landtaxvaluedollarcnt', 'taxamount',
                               'regionidzip', 'structuretaxvaluedollarcnt'])

    # Get month, year, weekday
    df.transactiondate = pd.to_datetime(df.transactiondate)
    df['transactionmonth'] = df['transactiondate'].dt.strftime('%Y%m')
    df['month'] = df.transactionmonth.str[4:]
    df['year'] = df.transactionmonth.str[:-2]
    df['weekday'] = df.transactiondate.dt.day_of_week

    df['year'] = df['year'].astype('int64')
    df['month'] = df['month'].astype('int64')

    # Select only certain features from full dataset
    X = df[['bedroomcnt', 'roomcnt', 'bathroomcnt', 'taxamount', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
            'latitude', 'longitude', 'month', 'year', 'weekday',
            'lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'yearbuilt',
            ]]
    Y = df.logerror

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=1234)

    return X_train, X_val, Y_train, Y_val, X, Y


def nn_train_val(X_train, X_val, Y_train, Y_val):

    # Applying standardization to inputs

    numeric_columns = ['bedroomcnt', 'roomcnt', 'bathroomcnt', 'taxamount',
                       'landtaxvaluedollarcnt', 'taxvaluedollarcnt',
                       'structuretaxvaluedollarcnt', 'latitude', 'longitude',
                       'lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'yearbuilt',
                       ]

    # Standardize numeric columns
    sc_x = StandardScaler()
    X_train_std = X_train.copy()
    X_val_std = X_val.copy()

    X_train_std[numeric_columns] = sc_x.fit(
        X_train[numeric_columns]).transform(X_train[numeric_columns])
    X_val_std[numeric_columns] = sc_x.fit(
        X_train[numeric_columns]).transform(X_val[numeric_columns])

    # Applying standardization to outputs
    Y_train_std = (Y_train - Y_train.mean())/Y_train.std()
    Y_val_std = (Y_val - Y_train.mean())/Y_train.std()

    # Mask missing data in columns - helps a tiny bit
    mask_value = 0
    X_train_std = X_train_std.fillna(mask_value)
    X_val_std = X_val_std.fillna(mask_value)

    # For conversions
    y_mean = Y_train.mean()
    y_std = Y_train.std()

    return X_train_std, X_val_std, Y_train_std, Y_val_std, y_mean, y_std


# Define the objective function
def objective_nn(trial, train_x, Y_train_std, val_x, Y_val_std):
    """Trial various hyperparameters for neural network model. Return validation MAE."""

    lr = trial.suggest_float("lr", 0.0005, 0.005)
    resolution_in_degrees = trial.suggest_float(
        "resolution_in_degrees", 0.05, 0.7)
    epochs = trial.suggest_int("epochs", 5, 10)
    batch = trial.suggest_int("batch", 4000, 10000)

    random.seed(42)
    tf.random.set_seed(1234)

    model = build_model(lr, resolution_in_degrees)

    # debug_input_shapes(train_x)
    # debug_input_shapes(val_x)
        # Train the model
    history = model.fit(
            x=train_x,
            y=Y_train_std,
            epochs=epochs,
            batch_size=batch,
            validation_data=(val_x, Y_val_std),
            verbose=0,)

    # Return the final validation MAE
    val_mae = history.history['val_mae'][-1]
    return val_mae


def build_model(lr, resolution_in_degrees):


    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    random.seed(42)

    bedroomcnt = layers.Input(shape=(1,), dtype=tf.float32, name='bedroomcnt')
    roomcnt = layers.Input(shape=(1,), dtype=tf.float32, name='roomcnt')
    bathroomcnt = layers.Input(shape=(1,), dtype=tf.float32, name='bathroomcnt')
    taxamount = layers.Input(shape=(1,), dtype=tf.float32, name='taxamount')
    landtaxvaluedollarcnt = layers.Input(shape=(1,), dtype=tf.float32, name='landtaxvaluedollarcnt')
    taxvaluedollarcnt = layers.Input(shape=(1,), dtype=tf.float32, name='taxvaluedollarcnt')
    structuretaxvaluedollarcnt = layers.Input(shape=(1,), dtype=tf.float32, name='structuretaxvaluedollarcnt')
    latitude = layers.Input(shape=(1,), dtype=tf.float32, name='latitude')
    longitude = layers.Input(shape=(1,), dtype=tf.float32, name='longitude')
    month = layers.Input(shape=(1,), dtype=tf.int64, name='month')
    year = layers.Input(shape=(1,), dtype=tf.int64, name='year')
    weekday = layers.Input(shape=(1,), dtype=tf.int64, name='weekday')


    lotsizesquarefeet = layers.Input(shape=(1,), dtype=tf.float32, name='lotsizesquarefeet')

    calculatedfinishedsquarefeet = layers.Input(shape=(1,), dtype=tf.float32, name='calculatedfinishedsquarefeet')


    yearbuilt = layers.Input(shape=(1,), dtype=tf.float32, name='yearbuilt')



    month_id = tf.keras.layers.IntegerLookup(
       vocabulary=list(range(1, 13)),
       output_mode='one_hot')(month)

    year_id = tf.keras.layers.IntegerLookup(
      vocabulary=[2016, 2017],
      output_mode='one_hot')(year)

    weekday_id = tf.keras.layers.IntegerLookup(
       vocabulary=[0,1,2,3,4,5,6],
       output_mode='one_hot')(weekday)

    # Create a list of numbers representing the bucket boundaries for latitude.
    latitude_boundaries = list(np.arange(-3, 3 + resolution_in_degrees, resolution_in_degrees))

    # Create a Discretization layer to separate the latitude data into buckets.
    latitude_discretized = tf.keras.layers.Discretization(
        bin_boundaries=latitude_boundaries,
        name='discretization_latitude')(latitude)

    # Create a list of numbers representing the bucket boundaries for longitude.
    longitude_boundaries = list(np.arange(-3, 3 + resolution_in_degrees, resolution_in_degrees))

    # Create a Discretization layer to separate the longitude data into buckets.
    longitude_discretized = tf.keras.layers.Discretization(
        bin_boundaries=longitude_boundaries,
        name='discretization_longitude')(longitude)

    # Cross the latitude and longitude features into a single one-hot vector.
    feature_cross = tf.keras.layers.HashedCrossing(
        num_bins=len(latitude_boundaries) * len(longitude_boundaries),
        output_mode='one_hot',
        name='cross_latitude_longitude')([latitude_discretized, longitude_discretized])

    features = layers.Concatenate()([
                    bedroomcnt,
                    roomcnt,
                    bathroomcnt,
                    taxamount,
                    landtaxvaluedollarcnt,
                    taxvaluedollarcnt,
                    structuretaxvaluedollarcnt,
                    feature_cross,
                    year_id,
                    month_id,
                    weekday_id,
                   lotsizesquarefeet,
                   calculatedfinishedsquarefeet,
                   yearbuilt,
    ])  


    x = layers.Dense(units=600, kernel_initializer='normal', activation='relu')(features)
    x = layers.Dropout(0.36)(x)
    x = layers.Dense(units=200, kernel_initializer='normal', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(1, kernel_initializer='normal')(x)
    x = layers.Dense(1, kernel_initializer='normal')(x)



    logerror = tf.keras.layers.Dense(
        units=1, activation='linear', name='logerror')(x)

    model = tf.keras.Model(inputs=[
        bedroomcnt,
        roomcnt,
        bathroomcnt,
        taxamount,
        landtaxvaluedollarcnt,
        taxvaluedollarcnt,
        structuretaxvaluedollarcnt,
        latitude,
        longitude,
        year,
       month,
       weekday,
       lotsizesquarefeet,
       calculatedfinishedsquarefeet,
       yearbuilt,
    ], outputs=logerror)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mae',
        metrics=['mae'])

    return model


######## Data reading specific to NN 
def read_test_data():
    sample = pd.read_csv('./data/sample_submission.csv')
    prop = pd.read_csv('./data/properties_2017.csv')
    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop, on='parcelid', how = 'left')

    return df_test, sample

def create_clean_test(df_test, X_train):

    train_columns =  ['bedroomcnt', 'roomcnt', 'bathroomcnt', 'taxamount',
        'landtaxvaluedollarcnt', 'taxvaluedollarcnt',
        'structuretaxvaluedollarcnt', 'latitude', 'longitude',
        'lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'yearbuilt',
        ]
    x_test = df_test[train_columns]

    # Set the transaction date dependent columns to constants
    x_test['month'] = 12
    x_test['year'] = 2016
    x_test['weekday'] = 4

    X_test_std = x_test.copy()

    # Scale-standardize features
    numeric_columns = ['bedroomcnt', 'roomcnt', 'bathroomcnt', 'taxamount',
        'landtaxvaluedollarcnt', 'taxvaluedollarcnt',
        'structuretaxvaluedollarcnt', 'latitude', 'longitude',
            'lotsizesquarefeet', 'calculatedfinishedsquarefeet', 'yearbuilt',
            ]
    sc_x = StandardScaler()
    sc_x.fit(X_train[numeric_columns])

    # Transform the test data using the same fitted scaler
    X_test_std[numeric_columns] = sc_x.transform(X_test_std[numeric_columns])

    # For latitude and longitude, we can't mask, so fill NAs with zeros
    X_test_std.longitude = X_test_std.longitude.fillna(0)
    X_test_std.latitude = X_test_std.latitude.fillna(0)

    # Mask missing data in columns 
    mask_value = -999
    X_test_std = X_test_std.fillna(mask_value)

    return X_test_std

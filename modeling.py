import numpy as np
import pandas as pd
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

random.seed(42)


"""
THIS WAS A TEST AND I HAVEN'T TESTED IT YEST. THE STUFF THAT RUNS IS IN 
nn_modeling.ipynb
"""

class RandomChoiceImputer(BaseEstimator, TransformerMixin):
    def __init__(self, choices):
        self.choices = choices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column in X.columns:
            missing_indices = X[column].isnull()
            num_missing = missing_indices.sum()
            if num_missing > 0:
                X.loc[missing_indices, column] = np.random.choice(self.choices, size=num_missing)
        return X

def load_data(base_path):
    print("Loading data...")
    properties_2016 = pd.read_csv(f'{base_path}properties_2016.csv', low_memory=False)
    properties_2017 = pd.read_csv(f'{base_path}properties_2017.csv', low_memory=False)
    train_2016 = pd.read_csv(f'{base_path}train_2016_v2.csv', low_memory=False)
    train_2017 = pd.read_csv(f'{base_path}train_2017.csv', low_memory=False)

    # Concatenate train datasets
    df_logs = pd.concat([train_2016, train_2017])

    # Concatenate properties datasets
    properties = pd.concat([properties_2016, properties_2017])

    # Merge train data with properties data
    df = pd.merge(df_logs, properties, on='parcelid', how='inner')

    # Convert transactiondate to datetime
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])

    return df

def preprocess_data(df, imputation_strategies):
    print("Preprocessing data...")
    train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=False)

    columns_to_keep = list(imputation_strategies.keys()) + ['transactiondate', 'logerror']
    train_df_selected = train_df[columns_to_keep].copy()
    valid_df_selected = valid_df[columns_to_keep].copy()

    imputers = {}

    # Apply imputation strategies to the training set
    for column, strategy in imputation_strategies.items():
        if strategy == 'drop':
            train_df_selected = train_df_selected.dropna(subset=[column])
        elif strategy == 'most_frequent':
            imputers[column] = SimpleImputer(strategy='most_frequent')
            train_df_selected[column] = imputers[column].fit_transform(train_df_selected[[column]])
        elif strategy == 'median':
            imputers[column] = SimpleImputer(strategy='median')
            train_df_selected[column] = imputers[column].fit_transform(train_df_selected[[column]])
        elif isinstance(strategy, (int, float)):
            imputers[column] = SimpleImputer(strategy='constant', fill_value=strategy)
            train_df_selected[column] = imputers[column].fit_transform(train_df_selected[[column]])
        elif strategy == 'random':
            unique_values = train_df_selected[column].dropna().unique().tolist()
            imputers[column] = RandomChoiceImputer(choices=unique_values)
            train_df_selected[[column]] = imputers[column].fit_transform(train_df_selected[[column]])

    # Apply IterativeImputer to the relevant columns in the training set
    iterative_columns = [column for column, strategy in imputation_strategies.items() if strategy == 'iterative']
    if iterative_columns:
        iterative_imputer = IterativeImputer()
        train_df_selected[iterative_columns] = iterative_imputer.fit_transform(train_df_selected[iterative_columns])

    # Apply the same imputation strategies to the validation set
    for column, strategy in imputation_strategies.items():
        if strategy == 'drop':
            valid_df_selected = valid_df_selected.dropna(subset=[column])
        elif strategy == 'most_frequent' or strategy == 'median' or isinstance(strategy, (int, float)):
            valid_df_selected[column] = imputers[column].transform(valid_df_selected[[column]])
        elif strategy == 'random':
            valid_df_selected[[column]] = imputers[column].transform(valid_df_selected[[column]])

    # Apply IterativeImputer to the relevant columns in the validation set
    if iterative_columns:
        valid_df_selected[iterative_columns] = iterative_imputer.transform(valid_df_selected[iterative_columns])

    return train_df_selected, valid_df_selected, imputers, iterative_imputer

def scale_and_encode_data(train_df_selected, valid_df_selected, numerical_features, categorical_features):
    print("Scaling and encoding data...")
    scalers = {}
    for column in numerical_features:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_df_selected[column] = scaler.fit_transform(train_df_selected[[column]])
        scalers[column] = scaler

    # Apply the same scaling to the validation set
    for column in numerical_features:
        valid_df_selected[column] = scalers[column].transform(valid_df_selected[[column]])

    encoders = {}
    for column in categorical_features:
        encoder = LabelEncoder()
        train_df_selected[column] = encoder.fit_transform(train_df_selected[column])
        encoders[column] = encoder
        
        # Handle unseen labels in validation set by encoding them to -1
        valid_df_selected[column] = valid_df_selected[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    return train_df_selected, valid_df_selected, scalers, encoders

def build_model(lr, input_features):
    print("Building model...")
    tf.keras.backend.clear_session()
    tf.random.set_seed(1234)
    random.seed(42)

    inputs = {feature: layers.Input(shape=(1,), dtype=tf.float32, name=feature) for feature in input_features}
    concatenated_features = layers.Concatenate()(list(inputs.values()))

    x = layers.Dense(units=1024, kernel_initializer='normal', activation='relu')(concatenated_features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=512, kernel_initializer='normal', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=256, kernel_initializer='normal', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=128, kernel_initializer='normal', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, kernel_initializer='normal')(x)

    logerror = layers.Dense(units=1, activation='linear', name='logerror')(x)

    model = models.Model(inputs=inputs, outputs=logerror)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mae', metrics=['mae'])

    return model

def train_model(model, X_train, Y_train_std, X_val, Y_val_std, numerical_features, categorical_features):
    print("Training model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x={feature: X_train[feature].values for feature in numerical_features + categorical_features},
        y=Y_train_std,
        epochs=100,
        batch_size=256,
        validation_data=(
            {feature: X_val[feature].values for feature in numerical_features + categorical_features},
            Y_val_std
        ),
        callbacks=[early_stopping]
    )

    return history

def evaluate_model(model, X_val, Y_train, Y_val, numerical_features, categorical_features):
    print("Evaluating model...")
    val_preds = model.predict({feature: X_val[feature].values for feature in numerical_features + categorical_features})
    val_preds = (val_preds[:, 0] * Y_train.std()) + Y_train.mean()

    def get_loss(y_true, y_pred):
        return tf.keras.losses.MAE(y_true, y_pred).numpy()

    mae = get_loss(Y_val, val_preds)
    print(f'Mean Absolute Error on Validation Set: {mae}')
    return mae

def prepare_test_data(test_df, month, columns_to_keep, imputation_strategies, imputers, iterative_imputer, scalers, encoders, numerical_features, categorical_features):
    test_df_selected = test_df.copy()
    
    # Add the month as transactiondate (this is a simplification)
    test_df_selected['transactiondate'] = pd.to_datetime(month, format='%Y%m')
    
    # Select relevant columns
    columns_to_keep_test = [col for col in columns_to_keep if col != 'logerror']
    test_df_selected = test_df_selected[columns_to_keep_test]
    
    # Impute missing values
    for column, strategy in imputation_strategies.items():
        if strategy == 'drop':
            test_df_selected = test_df_selected.dropna(subset=[column])
        elif strategy == 'most_frequent':
            test_df_selected[column] = imputers[column].transform(test_df_selected[[column]])
        elif strategy == 'median':
            test_df_selected[column] = imputers[column].transform(test_df_selected[[column]])
        elif isinstance(strategy, (int, float)):
            test_df_selected[column] = imputers[column].transform(test_df_selected[[column]])
        elif strategy == 'random':
            test_df_selected[[column]] = imputers[column].transform(test_df_selected[[column]])
    
    # Apply IterativeImputer to the relevant columns
    if iterative_columns:
        test_df_selected[iterative_columns] = iterative_imputer.transform(test_df_selected[iterative_columns])
    
    # Scale numerical features
    for column in numerical_features:
        test_df_selected[column] = scalers[column].transform(test_df_selected[[column]])
    
    # Encode categorical features
    for column in categorical_features:
        test_df_selected[column] = test_df_selected[column].apply(lambda x: encoders[column].transform([x])[0] if x in encoders[column].classes_ else -1)
    
    return test_df_selected

def generate_predictions(model, test_df, submission_df, columns_to_keep, imputation_strategies, imputers, iterative_imputer, scalers, encoders, numerical_features, categorical_features, Y_train):
    print("Generating predictions...")
    months = ['201610', '201611', '201612', '201710', '201711', '201712']

    for month in months:
        print(f"Processing month: {month}")
        test_df_selected = prepare_test_data(test_df, month, columns_to_keep, imputation_strategies, imputers, iterative_imputer, scalers, encoders, numerical_features, categorical_features)
        predictions = model.predict({feature: test_df_selected[feature].values for feature in numerical_features + categorical_features})
        predictions = (predictions[:, 0] * Y_train.std()) + Y_train.mean()
        submission_df[month] = predictions[:len(submission_df)]

    submission_df.to_csv('./data/sample_submission_updated2.csv', index=False)
    print('Submission file created successfully.')

def main():
    base_path = "./data/"
    df = load_data(base_path)

    imputation_strategies = {
        'bathroomcnt': 'most_frequent',
        'bedroomcnt': 'median',
        'calculatedbathnbr': 'median',
        'calculatedfinishedsquarefeet': 'median',
        'fireplacecnt': 0,
        'fullbathcnt': 'median',
        'garagecarcnt': 'iterative',
        'latitude': 'median',
        'longitude': 'median',
        'lotsizesquarefeet': 'iterative',
        'poolcnt': 0,
        'propertylandusetypeid': 'most_frequent',
        'regionidcounty': 'most_frequent',
        'regionidzip': 'drop',
        'roomcnt': 'drop',
        'yearbuilt': 'drop',
        'numberofstories': 0,
        'structuretaxvaluedollarcnt': 'iterative',
        'taxvaluedollarcnt': 'iterative',
        'assessmentyear': 'random',
        'landtaxvaluedollarcnt': 'iterative',
        'taxamount': 'iterative',
        'censustractandblock': 'random',
        'taxdelinquencyflag': 'N'
    }

    numerical_features = [
        'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet',
        'fireplacecnt', 'fullbathcnt', 'garagecarcnt', 'latitude', 'longitude', 'lotsizesquarefeet',
        'poolcnt', 'roomcnt', 'numberofstories', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
        'landtaxvaluedollarcnt', 'taxamount'
    ]

    categorical_features = [
        'propertylandusetypeid', 'regionidcounty', 'regionidzip', 'yearbuilt', 'assessmentyear',
        'censustractandblock'
    ]

    train_df_selected, valid_df_selected, imputers, iterative_imputer = preprocess_data(df, imputation_strategies)
    train_df_selected, valid_df_selected, scalers, encoders = scale_and_encode_data(train_df_selected, valid_df_selected, numerical_features, categorical_features)

    features = numerical_features + categorical_features
    target = 'logerror'

    X_train = train_df_selected[features]
    Y_train = train_df_selected[target]
    X_val = valid_df_selected[features]
    Y_val = valid_df_selected[target]

    Y_train_std = (Y_train - Y_train.mean()) / Y_train.std()
    Y_val_std = (Y_val - Y_train.mean()) / Y_train.std()

    model = build_model(lr=0.001, input_features=features)
    history = train_model(model, X_train, Y_train_std, X_val, Y_val_std, numerical_features, categorical_features)
    evaluate_model(model, X_val, Y_train, Y_val, numerical_features, categorical_features)

    # Load the sample submission file
    submission_df = pd.read_csv('./data/sample_submission.csv')
    
    # Load the properties data
    properties_2016 = pd.read_csv(f'{base_path}properties_2016.csv', low_memory=False)
    properties_2017 = pd.read_csv(f'{base_path}properties_2017.csv', low_memory=False)
    properties = pd.concat([properties_2016, properties_2017])
    
    # Merge with the base test data to get the necessary columns
    test_df_base = pd.DataFrame({'parcelid': submission_df['ParcelId'].values})
    test_df = pd.merge(test_df_base, properties, on='parcelid', how='left')
    
    generate_predictions(model, test_df, submission_df, columns_to_keep, imputation_strategies, imputers, iterative_imputer, scalers, encoders, numerical_features, categorical_features, Y_train)

if __name__ == "__main__":
    main()

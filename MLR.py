import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_properties = pd.read_csv('C:/Users/athen/Downloads/properties_2016.csv')

df_2016 = pd.read_csv('C:/Users/athen/Downloads/train_2016_v2.csv', parse_dates=["transactiondate"])
df_2017 = pd.read_csv('C:/Users/athen/Downloads/train_2017.csv', parse_dates=["transactiondate"])

df_logs = pd.concat([df_2016, df_2017])
df_logs

"""### Merge the main properties table with the properties transaction tables"""

df_all = pd.merge(df_logs, df_properties, on='parcelid', how='inner')

df_all

plt.figure(figsize=(12,12))
sns.jointplot(x=df_all.latitude.values, y=df_all.longitude.values, size=1)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()

(df_all.isnull().sum().sort_values(ascending=False))

df_all[df_all.regionidcounty.isnull()]

"""#### Remove all rows where latitude and longitude are missing"""

df_final = df_all[~df_all.regionidcounty.isnull()]

# The final missing values

(df_final.isnull().sum().sort_values(ascending=False))

"""1. We might be able to get regionidzip, regionidcity, censustractandblock - I'll work on this
2. The number of missing for garagearcnt and garagetotalsqft is the same, but there are explicit zeros? This is a little more tricky so we'll get back to this later
"""

print(df_final.fireplaceflag.value_counts())
print(df_final.threequarterbathnbr.value_counts())
print(df_final.buildingclasstypeid.value_counts())

print(df_final.poolcnt.value_counts())
print(df_final.storytypeid.value_counts())
print(df_final.typeconstructiontypeid.value_counts())

print(df_final.numberofstories.value_counts())
print(df_final.airconditioningtypeid.value_counts())
print(df_final.garagecarcnt.value_counts())

df_final[['garagetotalsqft']][df_final.garagetotalsqft==0.0]

"""Showing all properties where a property was sold more than once"""

v = df_final.parcelid.value_counts()
df_final[df_final.parcelid.isin(v.index[v.gt(2)])]

plt.hist(df_final.lotsizesquarefeet, bins=500)
plt.xlim(0,200000)

# Dropped row with null taxvaluedollarcnt
df_final = df_final.drop(df_final[df_final.taxvaluedollarcnt.isnull()].index)

"""Multi Linear Regression Model"""
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np


baseline_preds = np.ones(df_final.shape[0])*df_final['logerror'].mean()

# Getting column names of all columns w/o any missing values, and dropping parcelid
selected = df_final.columns[df_final.apply(lambda c: c.isnull().sum() == 0)].drop(['parcelid'])

# Getting selected columns
data = df_final[selected]

# Changing transactiondate to a datetime type
data['transactiondate'] = pd.to_datetime(data['transactiondate'])

# Extracting year based on the transaction date and setting it as a separate variable
data['year'] = data['transactiondate'].dt.year

# Setting test data to be all 2017 transactions
X_test = data[data['year'] == 2017]
y_test = X_test['logerror']
# Dropping logerror (outcome variable), transactiondate (represented by month and day),
# year and assessment year (since all observations in this subset have the same year value)
X_test = X_test.drop(['logerror', 'transactiondate', 'year', 'assessmentyear', 'fips'], axis=1)

# Setting train data to be all 2016 transactions
X_train = data[data['year'] != 2017]
y_train = X_train['logerror']
# Dropping logerror (outcome variable), transactiondate (represented by month and day),
# year and assessment year (since all observations in this subset have the same year value)
X_train = X_train.drop(['logerror', 'transactiondate', 'year', 'assessmentyear', 'fips'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardizing all features in X_train, X_val, and X_test
X_train_std = (X_train-X_train.mean())/X_train.std()
X_val_std = (X_val-X_train.mean())/X_train.std()
X_test_std = (X_test-X_train.mean())/X_train.std()

# Standardizing Y_train, Y_val, and Y_test
y_train_std = (y_train-y_train.mean())/y_train.std()
y_val_std = (y_val-y_train.mean())/y_train.std()
y_test_std = (y_test-y_train.mean())/y_train.std()

def build_model(num_features, learning_rate):
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
  optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

  # Finally, compile the model. This finalizes the graph for training.
  # We specify the loss and the optimizer above
  model.compile(
        optimizer=optimizer,
        loss='mae'
  )

  return model

# Build and compile test_model
test_model = build_model(num_features=X_train.shape[1],learning_rate=0.0004)

# Fit test model
test_num_epochs=10
test_train_tf = test_model.fit(x=X_train_std, y=y_train_std, epochs=test_num_epochs, verbose=0,
                         validation_data=(X_val_std, y_val_std))

# Plotting losses of test model
plt.scatter(np.arange(1, test_num_epochs+1), test_train_tf.history['loss'], label="Training loss")
plt.scatter(np.arange(1, test_num_epochs+1), test_train_tf.history['val_loss'], label="Validation loss")
plt.xticks(np.arange(1, test_num_epochs+1, 1))
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend(loc=(1.05, 0.5))

# Printing out learned model parameters, final epoch loss for training
# and validation data, and percentage difference between
# losses on training and validation data
print("Learned model parameters:", test_model.layers[0].get_weights())
print("\nFinal epoch loss on training data:", test_train_tf.history['loss'][-1])
print("\nFinal epoch loss on validation data:", test_train_tf.history['val_loss'][-1])
print("\nPercentage difference between the losses:",
 ((np.array(test_train_tf.history['loss']) - np.array(test_train_tf.history['val_loss']))
 / np.array(test_train_tf.history['val_loss'])) * 100)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_preds = test_model.predict(X_train_scaled)
test_preds = test_model.predict(X_test_scaled)

print("Train MAE:", mean_absolute_error(y_train, train_preds))
print("Test MAE:", mean_absolute_error(y_test, test_preds))
print("Baseline Train MAE:", mean_absolute_error(y_train, baseline_preds[:len(y_train)]))
print("Baseline Test MAE:", mean_absolute_error(y_test, baseline_preds[:len(y_test)]))
